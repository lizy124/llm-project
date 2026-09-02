# PR #15367 服务器 e2e 验证详细报告 — rebase 复测(2026-09-01)

- 验证对象:vllm-ascend 分支 `refactor_layerwise_part1` @ `63be9e03b2f7a5e2af6bc13bad6617a817da7d8c`(rebase 到 upstream/main `72a988f9d` 之后的 12 个 PR commits)
- 配对 vllm:@ `ba07e4a48fc951300d97eb506217dd530583dea3`(与首轮一致,未动)
- 执行环境:192.168.13.165 / Docker 容器 `refactor_165`
- 执行日期:2026-09-01 18:31–18:57(UTC+8),编排脚本 `test/rerun_e2e.sh` 后台执行
- 结论:**三项验证全部 PASS**
- 上一轮报告:`e2e-report-20260901.md`(对象 `2a239d18a`,检视返工 rebase 前基线;两轮均 PASS,行为一致性互证)
- 证据归档:服务器 `map_165/record_final/*_20260901_rebase/` + `rerun_e2e.log`

---

## 1. 复测背景

首轮验证(`e2e-report-20260901.md`)基于 `2a239d18a`。其后发生了两类变化,原报告过期,需在新代码上复测:

1. **检视返工(4 commits,`f057d367c..5c550766d` → rebase 后 `108a8d8a9..63be9e03b`)**:GVA 协议四函数迁入 memcache backend、通用层改名 `use_layerwise_transfer`、删 `gva_protocol.py`、layerwise 线程后端无关化(删 `MemcacheBackend` assert)
2. **mypy 修复**(`5c550766d`):`backend_map` 显式注解
3. **rebase 到最新 main**(`e8f47fc11` → `72a988f9d`):带入 upstream KV Pool 改动,尤其 #15364(Avoid prompt token copies when KV events are disabled),与池化路径有交集

### 验证清单与验收标准(与首轮一致)

| # | 验证项 | 目的 | 验收标准 |
|---|---|---|---|
| 1 | MultiConnector PD 分离 | #14465 修复生效点实测 | 请求成功率 100%;初始化链路无 AttributeError |
| 2 | memcache layerwise 冒烟 | load 路径非零 gva,排除静默失效 | `load_gvas valid_gvas>0`;`hit_check hit_tokens>0` |
| 3 | mooncake 非 layerwise 冒烟 | 通用路径零回归 | 与 main 基线行为一致;三维证据链通过 |

## 2. 环境指纹

- 拓扑/模型/池化服务/启动参数与首轮完全一致(DSV2-Lite-Chat + Qwen3-32B、MetaService :8000、mooncake master :9008、公共参数七条铁律),详见首轮报告 §2,此处不重复
- 版本差异仅 vllm-ascend:`63be9e03b`(git rev-parse 证实;`pip list` 显示 `+g2a239d18a` 为 editable 构建版本串滞后,非实际代码版本)
- 部署方式:服务器到 GitHub 间歇断连,改走 git bundle(`rerun.bundle`,`refactor_layerwise_part1 ^e8f47fc11`)scp + docker cp + `git fetch <bundle>` + `git reset --hard 63be9e03b` + `pip install -e .`

## 3. 场景 2:memcache layerwise — PASS

配置与首轮一致(DSV2-Lite TP=4 :8004,`backend=memcache` + `use_layerwise=true`)。

**① layerwise 激活(配置面)**

```
(Worker_TP3) INFO 09-01 18:44:52 [pool_worker.py:442] layerwise config: num_layers=27 num_groups=1
```

(行号 445→442:rebase 带入 upstream 代码导致偏移,内容不变)

**② hit_check hit_tokens>0(验收标准)**

```
(EngineCore) DEBUG [pool_scheduler.py:388] hit_check: req=chatcmpl-b6543338... token_len=3457 hits_per_group=[3328] hit_tokens=3328
```

**③ load_gvas valid_gvas>0(验收标准)**

```
(Worker_TP0/1/3) DEBUG [pool_worker.py:1503] load_gvas: req=chatcmpl-b6543338... group=0 eff_bs=128 load_blocks=[0,26) keys=26 valid_gvas=26 lease_fail=0
```

**④ MetaService 三维证据链**:`alloc_successes=28 stored_keys=28 query_successes=130 query_not_found=28`

**⑤ vllm /metrics**:`external_prefix_cache_hits_total = 3328`

**⑥ 致命错误扫描**:无 Traceback / Segfault。

与首轮差异:hit 3328(26 块)vs 首轮 3456(27 块)——第二发请求时最后一块未进滑动窗口,块对齐时机差异,非缺陷;判据(>0)不变。首轮首发 1.13s→二发 0.16s、本轮 0.33s→1.40s,elapsed 为观察项非判据(两轮 completion 长度不同:2 vs 16 tokens)。

## 4. 场景 3:mooncake 非 layerwise 零回归 — PASS

配置与首轮一致(Qwen3-32B TP=4 :8006,`backend=mooncake`,无 `use_layerwise`)。

**① 无 layerwise 痕迹(零回归判据)**:server.log 全文 grep `load_gvas:` / `hit_check:` 均 0 行。

**② master 三维证据链**:`allocated_bytes=939524096 key_count=112 active_clients=4`(939.5MB / 112 keys / 4 pool clients,与首轮同量级)

**③ vllm /metrics**:`external_prefix_cache_hits_total = 3328`(第二发经池命中)

**④ 致命错误扫描**:无 Traceback / Segfault / Initialize mooncake failed。

## 5. 场景 1:MultiConnector PD 分离(#14465 回归点)— PASS

拓扑与首轮一致:proxy :9000 → P :8100(DSV2-Lite TP=4 chips 0-3,MooncakeLayerwise + AscendStore/memcache layerwise 双 connector)→ D :8200(TP=4 chips 4-7,MooncakeLayerwise consumer)。

**① 初始化链路无 AttributeError(验收标准)**

`Creating v1 connector with name: AscendMultiConnector` ×5(4 workers + EngineCore);P/D 日志 grep `AttributeError` = 0 行。检视返工的关键路径(`set_external_slot_release_waiter` 经 worker 侧 gate 下沉后)完整走通。

**② 请求成功率 100%(验收标准)**

```
req0 ok elapsed=0.66s prompt_tokens=4177 completion=2   (合成共享前缀+Q A)
req1 ok elapsed=0.47s prompt_tokens=4177 completion=2   (合成共享前缀+Q B)
req2 ok elapsed=0.45s prompt_tokens=4177 completion=2   (合成共享前缀+Q C)
req3 ok elapsed=2.92s prompt_tokens=4236 completion=32  (GSM8K-lite 真实问题 1)
req4 ok elapsed=2.76s prompt_tokens=4198 completion=32  (GSM8K-lite 真实问题 2)
success_rate=5/5
```

**③ 传输链路证据(P 侧)**:`load_gvas` 16 行(AscendStore layerwise 池写活跃);MooncakeLayerwise 按层派发,D 侧 27 层(`model.layers.0`–`26`)LayerMetadata(基地址/block_len)完整经 metaserver 到达 P,`remote_engine_id='1-...'` 与 D 配置一致。

**④ 传输链路证据(D 侧)**:recv 相关 34 行,与 P 侧发送对应。

**⑤ P 侧 pool 证据**:MetaService `query_successes=770`。put 命名指标不存在(memcache 1.2.0 已知限制,与首轮同形态 WARN;池写由 load_gvas 行证实)。

**⑥ 致命错误扫描**:P/D 两侧均无 Traceback / Segfault。

与首轮差异:load_gvas 16 行(首轮 28)、recv 34 行(首轮 52)——请求数相同(5 发)下行数随命中窗口分布浮动,链路证据行 > 0 判据不变。

## 6. 结论

| 验证项 | 结果 | 关键证据 |
|---|---|---|
| 1. MultiConnector PD 分离 | **PASS** | 5/5(100%);无 AttributeError;AscendMultiConnector ×5;load_gvas=16 行;recv=34 行 |
| 2. memcache layerwise | **PASS** | hit_tokens=3328>0;valid_gvas=26>0, lease_fail=0;alloc=28/stored=28/query=130 |
| 3. mooncake 非 layerwise | **PASS** | 无 layerwise 标记;master 三维 939.5MB/112 keys/4 clients;external hits=3328 |

检视返工 4 commits + mypy 修复 + rebase 到 `72a988f9d`(含 #15364 KV Pool 改动)后,三项验证与首轮(返工前基线 `2a239d18a`)全部通过,行为一致。两轮互证:返工为行为保持的重构,通用路径与 layerwise 路径均无回归。

过程问题仅 1 项(测试基建):服务器到 GitHub 间歇断连导致 `git fetch` 两次失败,改走 git bundle 部署(见 §2),被测代码零改动。
