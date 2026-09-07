# 预优化基线组测试方法（Layerwise 池化优化前性能摸底）

> 需求：SR20260820223202（Layerwise KV Cache Pooling 优化，见 [../01_requirements_analysis.md](../01_requirements_analysis.md)）
> 范围：当前这套栈在**未做本课题 layerwise 优化**时的性能摸底（优化前对照）；优化后另测一版对比
> 服务器：192.168.13.165（A3 × 16 卡 / 910B3 / 每芯 64G HBM）
> 容器：`pool_165`（镜像自带 `/vllm-workspace`，vllm + vllm-ascend 均 editable 指向源码树）
> 权重：`/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp`（W8A8 量化 + MTP draft）
> 栈（实测）：vllm upstream `33e849499` / vllm-ascend `33e849499`，**未含** PR #15442 / #15854
> 编制：2026-09-07（首版，随进展修订）

---

## 一、目标与产出

| # | 产出 | 用途 |
|---|---|---|
| O1 | layerwise 池化（优化前）单/多前缀 TPS、TTFT | 优化前后对照主指标（同事表格缺 TPS/TTFT，正是要补的） |
| O2 | KV 池命中率（device_sdma / memcache） | 验证池化生效，区分"优化点" |
| O3 | MTP 采纳率（若启用） | 特性叠加基线 |
| O4 | HBM PrefixCache 对照组数据 | 池化 vs HBM 的收益判断（见 `../baseline/`） |
| O5 | 可复现基线：脚本 + 配置 + 数据归档 | 优化后同口径重测对比 |

## 二、目标口径（对齐同事，见 [../07_colleague_test_data.md](../07_colleague_test_data.md)）

| 项 | 同事值 | 165 当前可达 |
|---|---|---|
| TP×DP | TP4×DP4（16 卡） | TP8×DP1（8 卡）**待修复后恢复** |
| max-model-len | 1048576 | 131072（1M 时 TP8 每卡 KV 不足） |
| MTP | 开 | 关 |
| 并发 | 32 | 32 |
| 输入 | 单前缀 1×131072 / 多前缀 32×131072 | 同上 |
| 输出 | 1 | 1 |
| kv-transfer | AscendStoreConnector(memcache, use_layerwise: true, prefetch_layers: 3) | 当前仅 use_layerwise: false 可跑通 |

**结论：165 栈下同事口径不可复现（DP4 死锁 + layerwise 死锁，见 stage_summary）。基线先按变体口径采集，并记录差异。**

## 三、环境

- 容器：`pool_165`；MetaService（memcache_hybrid）需保持存活（启动脚本会检查/拉起）
- 大页：2MB hugepage 200000 页（400G）；KV 池 DRAM 16G（HalMemCreate 128G 会失败 ret:6）
- 本地 ⇔ 服务器：`D:\lzy\project\kv_pool\tmp\*.sh` → scp → 165 `/home/lizhongyang/tmp/` → `docker cp` → 容器 `/home/lizhongyang/tmp/`
- 日志：容器内 `/home/lizhongyang/map_165/run/dsv4_layerwise_v2_8100.log`（每轮 launch 前自动备份）

## 四、指标定义

| 指标 | 口径 | 来源 |
|---|---|---|
| Prefill TPS | 总输入 token / 压测时长 | benchmark_serving.py / 自研压测脚本 |
| TTFT | mean / p99 | 同上 |
| 命中率 | KV 池命中（token 级） | kv pool 日志 / /metrics |
| MTP 采纳率 | 采纳 token / 提议 token | MTP 日志 |

## 五、当前可运行服务配置（变体口径）

```
TP8 × DP1, max_model_len 131072, max-num-batched-tokens 10240, max-num-seqs 64
--no-enable-prefix-caching --enforce-eager --async-scheduling
--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
--additional-config {enable_npugraph_ex, enable_cpu_binding, enable_dsa_cp, enable_flashcomm1(已弃用)}
--kv-transfer-config AscendStoreConnector(memcache, use_layerwise: false)
```

启动脚本：`run/start_dsv4_layerwise.sh`（参数位于文件头，每次改动同步更新本表）。

## 六、执行步骤

```
S0  确保 MetaService 存活 + 大页已配置 → launch_16c_v2.sh 清理残留并启动
S1  起服务 → 等 APIServer READY + EngineCore idle
S2  verify_dp1.sh：单请求 + 32 并发冒烟
S3  前缀灌入（预热）→ 单/多前缀压测 → 记录 TPS/TTFT/命中率
S4  数据归档 pre_optimization_baseline/data/，更新 report.md
```

## 七、控制变量

1. 栈版本：editable 代码冻结，禁止 git pull/checkout；每轮归档 `git rev-parse HEAD`
2. 数据可复现：输入 seed 固定、启动命令与压测命令全量归档
3. 编译/预热抖动：压测前先 warmup
4. 口径一致性：优化前后必须同栈同配置对比
