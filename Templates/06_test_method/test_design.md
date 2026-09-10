## 5. 池化测试方法

> 如何设计测试请求、如何组织验证矩阵、如何做对照实验。
> 参考脚本：`map_51/pr14465_dsv4_kvpool/test/`（auto_verify_clean.sh / verify_load_new.sh / multi_growth_requests.sh）。

### 5.1 长 prompt 门槛（最常踩的坑，先算再发）

**存入条件**：`num_tokens_to_save ≥ chunk_boundary`，其中
`cache_transfer_granularity = lcm_block_size`（DSV4 = **4096**）。

| 模型/场景 | granularity | prompt 要求 |
|-----------|-------------|-------------|
| DSV4-Flash（PR14465） | 4096 | **≥ 2 个完整 4096 块 → ≥ 8192 token** |
| PR14912 验证场景 | 128 | ≥ 5042 token（实测门槛） |

规则：**先确认目标模型的 cache_transfer_granularity，prompt ≥ 2 个完整块**。
短 prompt（如 629 token）→ `num_tokens_to_save=0` → `skip_save=True` → 整个请求不存入，
master 指标全 0，极易误判为"池化未生效"。

实用做法：基础段落重复拼接。参考 auto_verify_clean.sh——
~182 token 的英文段重复 260 次 ≈ 11.8K token（`num_tokens_to_save=8192 > 4096`），稳定触发存入。

**注意块对齐**：仅长度超过还不够，块 hash 需对齐（`available_full_block_count = len(block_hashes) // hashes_per_transfer_block`）。
PR14465 首跑 3 条 ~13K prompt 全 0 就是对齐问题，改用规整重复文本后成功。

### 5.2 验证矩阵设计（backend × layerwise）

池化 PR 验证的标准矩阵（PR14465 模式，4 组）：

| # | backend | use_layerwise | DSV4 预期 |
|---|---------|---------------|-----------|
| 1 | mooncake | false | PASS（核心路径） |
| 2 | mooncake | true | EXPECTED_FAIL（DSV4 NSA 硬限） |
| 3 | memcache | false | PASS |
| 4 | memcache | true | EXPECTED_FAIL（同根因） |

要点：
- **先跑 non-layerwise**（核心功能），PASS 后再碰 layerwise
- layerwise 失败若为 multi-spec 硬限 → 判 **EXPECTED_FAIL / BLOCKED**，不算 PR 缺陷（见 §8.1）
- 同一矩阵内**干净基线**：每组开始前全清重启（kill vllm + master → counters 归零），参考 `full_restart_clean.sh`

### 5.3 标准测试流程（存 → 取 → 去重）

```
0. （推荐）**warm-up 预热**：ASCEND_BUFFER_POOL 场景下 HCCL one-sided 连接懒建（官方 kv_pool §2.4.2，
   每连接 4MB device mem），压测前先发 2 个单请求预热（官方建议：input 8K / output 1 / 总请求数 2–3× 设备数）
   ——消除首次连接耗时与断连对指标/成功率的污染
1. 等 READY（status.txt 出现 PASS: service ready）
2. 记录基线 counters：
   curl -s http://127.0.0.1:9008/metrics | grep -E '^master_(allocated_bytes|key_count|active_clients) '
   → 干净基线应为 0 / 0 / DP×TP
3. 【存】并发发 N 个长 prompt（N=20，curl 后台并发），max_tokens 64 即可（省时间）
   → 请求完 + 15s 后再采样 counters，应增长
4. 【取+去重】用完全相同的内容重发若干次（verify_load_new.sh 模式）
   → key_count / allocated 应恒定不增长（去重）
   → server.log `External prefix cache hit rate` 应 > 0（取）
5. 【增长】发不同内容长 prompt ×3（multi_growth_requests.sh 模式）
   → keys 应阶梯增长（61 → 244 这类）
6. 全程采样 master.log 的 Batch Requests 速率行（PutStart/Get/ExistKey 的 Req/Item 每秒速率）
```

并发参考：轮次递增设计（3 → 50 → 30 → 30 不同内容 → 30 不同+主题段），
第一轮小并发先验证链路，再加压。

### 5.4 对照实验设计（控制变量定位根因）

当"存取不发生"时，用标准 attention 模型做对照（PR14465 的 Qwen3-8B 实验）：

- 相同 mooncake 配置，只换模型（DSV4 hybrid attention ↔ Qwen3 标准 attention）
- Qwen3 能存取 → 排除 `--no-enable-prefix-caching` / 权重 / 环境问题 → 根因在模型特性
- 实测结论：DSV4 早期全 0 的根因是 granularity=4096 + 短/未对齐 prompt 的 skip_save，不是 prefix-caching

**原则：怀疑某配置是根因时，固定其他变量单独翻转它**（用户偏好的控制变量比对法）。

### 5.5 layerwise 路径的额外观察点

layerwise（`use_layerwise=true` + `layerwise_prefetch_layers`）验证时额外看：

1. 启动期：`build_layerwise_reuse_layout` 是否抛 multi-spec ValueError（DSV4 必抛，见 §8.1）
2. 运行期：kv_transfer 日志的 layer load 事件时间戳（PR14912 的 `_TimedLayerLoadEvent` 语义）
3. 指标口径与 sync 不同（load_keys = blocks×layers，放大 64 倍，**跨 path 不可直接对比**，见 §8.4）
4. layerwise 存入时机在请求完成后（non-layerwise 亦然），不要期望请求中途看到 PutStart
5. **layerwise 激活面（配置证据）**：worker 日志 `layerwise config: num_layers=N num_groups=M`——没有这行说明 gate 没开（backend 未 opt-in / 参数没传到）
6. **layerwise 判据的正计数形态（PR15367 实测，memcache）**：
   - 存侧：`load_gvas: req=... keys=N valid_gvas=N lease_fail=0`（keys>0 即入池；lease_fail=0 排除租约失败假阳性）
   - 取侧：`hit_check: req=... token_len=N hits_per_group=[N] hit_tokens=N`（hit_tokens>0 即命中）
   - 注意首行 `hit_check ... token_len=15 no participating groups` 是预热小请求的正常跳过，判据用 `hit_tokens=` 的行（长前缀请求）

### 5.5b PD 分诊（kv_producer/kv_consumer + MultiConnector + proxy）观察点（PR15367 S1，单机 8 卡实测）

PD 分诊拓扑：proxy :9000 → P :8100（TP=4，MooncakeLayerwise + AscendStore/memcache layerwise 双 connector）→ D :8200（TP=4，MooncakeLayerwise consumer）。判据（全部正计数）：

| 判据 | 证据形态 | 封堵的失效 |
|------|---------|-----------|
| 成功率 100% | `success_rate=5/5`，每请求带回真实 `prompt_tokens=N completion=M`（死服务给不出带 token 计数的 completion） | 服务没起来 |
| connector 初始化无异常 | P/D 日志 `Creating v1 connector with name: AscendMultiConnector` ×N，grep `AttributeError` = 0 | 初始化路径崩（waiter 下沉类重构的关键回归点） |
| P 侧池写活跃 | prefill.log `load_gvas:` 行数 > 0 + MetaService `query_successes` > 0 | 起来了但没入池 |
| D 侧接收完整 | decode.log layerwise recv 行数 > 0 + **27 层 LayerMetadata（`model.layers.0`–`26`）完整经 metaserver 到达 P**，`remote_engine_id` 与 D 配置一致 | KV 传输链路断裂/静默丢层 |

要点：LayerMetadata 逐层数（0 到 num_layers-1 连续）是"传输内容完整"的强判据；P 侧发送行数与 D 侧接收行数应同量级（浮动属正常，判据是 >0）。

### 5.6 测试脚本纪律

- 复杂流程写 .sh push 到远端跑（PowerShell → ssh 引号转义不可靠，见 AGENTS.md §1）
- 长 prompt 不要硬编码在命令行里，脚本内拼接
- 每轮测试的输出重定向到日志文件并保留（验证过程日志必须留档）
- 请求体存 `/tmp/creq_$n.json` 便于事后检查响应内容是否正常
- **服务器到 GitHub 断连时的代码部署走 git bundle**：本机 `git bundle create f.bundle '分支名' '^基线'`（⚠️ PowerShell 会吞 `..` range 且裸 hash 不产生 ref 条目，必须引号分支名 + ^排除式）→ scp 宿主机 → docker cp 进容器 → `git fetch /tmp/f.bundle 'refs/heads/b:refs/heads/from-bundle'` → `git reset --hard <sha>`；fetch 失败一次先 timeout 60 重试再降级 bundle
- **e2e 多场景编排在宿主机 nohup 跑总脚本**（更新代码 → clean → 前置服务 → 逐场景 start/test/stop → clean），单场景失败不中断后续场景；场景间必须 stop + clean_npu，否则 HBM 残留污染下一场景

---

