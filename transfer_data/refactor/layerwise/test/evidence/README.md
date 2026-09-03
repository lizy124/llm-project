# evidence — PR #15367 服务器验证完整材料

> 别人拿到这个目录应能**完全复现**本次验证：每个场景用什么环境变量、哪个脚本拉起服务、
> 哪个脚本发请求和判定、结果证据长什么样。通用方法论（为什么这样设计判据）见
> [../../../playbook/verify_guide.md](../../../playbook/verify_guide.md)；逐场景操作手册见
> [../../../playbook/run_dir/](../../../playbook/run_dir/)。

## 验证对象与结论

- 分支：`lizy124/vllm-ascend:refactor_layerwise_part1`，两轮验证：
  - 轮 1（基线）：`2a239d18a`（检视返工前）→ 三场景 PASS
  - 轮 2（复测）：`63be9e03b`（返工 4 commits + mypy 修复 + rebase 到 upstream/main 后）→ 三场景 PASS
  - 两轮同构 PASS 互证重构行为不变（报告：[../e2e-report-20260901.md](../e2e-report-20260901.md) /
    [../e2e-report-20260901-rebase.md](../e2e-report-20260901-rebase.md)）
- 服务器：192.168.13.165 / 容器 `refactor_165`（单机 8 卡）
- 服务器侧归档：`/home/lizhongyang/map_165/record_final/`（含 `*_20260901` 与 `*_20260901_rebase` 两轮）

## 目录结构

```
evidence/
├── README.md                ← 本文档(总说明)
├── scripts/                 ← 全部可执行脚本(服务器实测原样拷贝)
├── s1_pd_multiconn/         ← 轮 1 证据:MultiConnector PD 分离(状态/环境快照)
├── s2_memcache_layerwise/   ← 轮 1 证据:memcache layerwise
└── s3_mooncake_non_layerwise/ ← 轮 1 证据:mooncake 非 layerwise 零回归
```

轮 2 证据未同步到本地（体量大），在服务器 `record_final/*_20260901_rebase/`；
轮 1 证据摘要也在各场景 `*_status.txt`，完整 test.log 在服务器。

## scripts/ 分类与用途

### 前置准备（宿主机/容器混合，先跑）

| 脚本 | 执行位置 | 干什么 |
|------|---------|--------|
| `pool_prep.sh` | 宿主机 | memcache 一键前置：hugepages(200000) → mmc-local.conf 改 `device_sdma`+dram 16GB → 拉起 MetaService → 探活 5000/6000 |
| `start_metaservice.sh` | 容器内 | MetaService 单独拉起（pool_prep 第 3 步的实体） |
| `start_mooncake_master.sh` | 容器内 | mooncake master 拉起（50088/9008，显式 PID） |
| `mooncake.json` | — | mooncake master 配置（P2PHANDSHAKE / ascend / 5GB segment） |

### 场景启动（容器内）

| 脚本 | 场景 | 关键内容 |
|------|------|---------|
| `s2_start.sh` | S2 | DSV2-Lite + memcache layerwise（:8004, TP=4, chips 0-3, `use_layerwise=true`）；内含全部 11 个环境变量与 kv-transfer-config |
| `s3_start.sh` | S3 | Qwen3-32B + mooncake 非 layerwise（:8006, TP=4, 无 use_layerwise） |
| `s1_start_prefill.sh` | S1-P | DSV2-Lite :8100（chips 0-3），MultiConnector 双 connector（MooncakeLayerwise + AscendStore/memcache layerwise），engine_id=0 |
| `s1_start_decode.sh` | S1-D | DSV2-Lite :8200（chips 4-7），MooncakeLayerwise kv_consumer，engine_id=1 |
| `s1_start_proxy.sh` | S1-proxy | layerwise 分诊代理 :9000（examples/load_balance_proxy_layerwise_server_example.py） |

启动脚本共同模式：参数/环境变量全在脚本头部（`set -Eeuo pipefail`）→ 预检（模型存在/端口空闲）→
nohup 拉起 + 日志/PID/env 快照落盘 → 轮询 `/v1/models` 就绪 → `PASS: service ready` 写入 status.txt。

### 场景判定（容器内，服务就绪后跑）

| 脚本 | 判什么 |
|------|--------|
| `s2_test.sh` | layerwise 激活行 + `hit_tokens>0` + `valid_gvas>0` + vllm external hits + MetaService 三维（alloc/stored/query/not_found）+ 致命错误扫描 |
| `s3_test.sh` | 请求序列 + **layerwise 标记必须缺席**（零回归）+ mooncake master 三维 + external hits + 错误扫描 |
| `s1_test.sh` | P/D 零 AttributeError（#14465 回归点）+ 经 proxy 请求 5 发 100% 成功（3 合成共享前缀 + 2 GSM8K-lite 真实问题）+ P 侧 load_gvas + D 侧 recv + MetaService put/query |

### 编排与清理

| 脚本 | 用途 |
|------|------|
| `rerun_e2e.sh` | **轮 2 总编排**（宿主机 nohup）：更新代码（git fetch 或 bundle）→ clean → pool_prep → S2 → S3(含 master) → S1 → 汇总各场景 status |
| `stop_server.sh` / `stop_s1.sh` | 停单实例 / 停 PD 三进程（proxy→decode→prefill） |
| `clean_npu.sh` | 清残留 vllm 进程 + 确认 NPU 释放（场景间必跑） |

### 其余（排障期工具，非主链路）

`archive.sh`（归档）、`collect_status.sh`/`tail_s1.sh`/`show_s1_result.sh`（状态查看）、
`find_metaservice.sh`/`recon_*.sh`/`rootcause_s1.sh`（问题定位）、`probe_weights*.sh`（权重探查）、
`npu_mem.sh`（HBM 查看）。

## 完整执行序列（复现步骤）

```bash
# 宿主机(ssh root@192.168.13.165)
docker exec refactor_165 bash -c "cd /vllm-workspace/vllm-ascend && git fetch fork refactor_layerwise_part1 && git reset --hard 63be9e03b && pip install -e . --no-deps --no-build-isolation -q"
# 或:服务器连不上 GitHub 时走 bundle(见 scripts/rerun_e2e.sh [1/6] 的 bundle 分支)

bash /home/lizhongyang/map_165/test/rerun_e2e.sh    # 或逐场景手动执行(见下)

# 逐场景手动模式(容器内):
bash $BASE/start/pool_prep.sh                        # 前置(memcache 路径)
docker exec refactor_165 bash $BASE/test/s2_start.sh && docker exec refactor_165 bash $BASE/test/s2_test.sh
docker exec refactor_165 bash $BASE/test/stop_server.sh $BASE/run/s2_memcache_layerwise && docker exec refactor_165 bash $BASE/test/clean_npu.sh
docker exec refactor_165 bash $BASE/start/start_mooncake_master.sh   # mooncake 前置
docker exec refactor_165 bash $BASE/test/s3_start.sh && docker exec refactor_165 bash $BASE/test/s3_test.sh
docker exec refactor_165 bash $BASE/test/stop_server.sh $BASE/run/s3_mooncake_non_layerwise && docker exec refactor_165 bash $BASE/test/clean_npu.sh
docker exec refactor_165 bash $BASE/test/s1_start_prefill.sh && docker exec refactor_165 bash $BASE/test/s1_start_decode.sh && docker exec refactor_165 bash $BASE/test/s1_start_proxy.sh
docker exec refactor_165 bash $BASE/test/s1_test.sh
docker exec refactor_165 bash $BASE/test/stop_s1.sh && docker exec refactor_165 bash $BASE/test/clean_npu.sh
```

各场景环境变量、kv-transfer-config JSON、请求构造的逐行说明：`scripts/` 对应脚本内注释 +
[../../../playbook/run_dir/](../../../playbook/run_dir/) 四篇场景手册（含本 PR 实测的
"通过形态示例"数值）。

## 证据目录说明（轮 1 本地镜像）

- `*/env.txt`：启动时快照（日期/模型/TP/chips/git rev-parse 双仓/pip 版本）——"验证的到底是哪个 commit"的证据
- `*/status.txt`：服务就绪状态（`PASS: service ready at ...`）
- `*/test_status.txt`：判定结论（`PASS: S2 ...` + 判据摘要）
- `s1_pd_multiconn/`：另有 `prefill_multiconn_init.txt`（MultiConnector 初始化行）、
  `prefill_load_gvas_count.txt`（load_gvas 行数）、`metaservice_metrics.txt`（MetaService 计数）
- `s2_memcache_layerwise/layerwise_markers.txt`：`layerwise config` / `load_gvas` / `hit_check` 原始行
- `s3_mooncake_non_layerwise/`：`pool_metrics.txt`（master 三维）+ `vllm_metrics.txt`（external hits）

## 判据设计依据

判据全部是"正计数 + 跨进程证人"（详见 playbook/verify_guide.md §6.6）：
- 服务没起 → 请求 usage 带不回真实 token 数
- 没入池 → MetaService/mooncake master 自己进程的计数为 0
- 静默不命中 → `hit_tokens=0`、`external_prefix_cache_hits_total=0`
- 方法有效性 → S2（layerwise 标记 >0）与 S3（同一 grep = 0）构成阴性对照对
