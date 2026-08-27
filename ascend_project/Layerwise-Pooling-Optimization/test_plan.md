# Layerwise 池化优化 — 第一轮测试计划（165 服务器）

> 被测对象：`layerwise_pooling` 分支（D1 整合 + D2 not overlapped 指标，见 [dev_plan.md](dev_plan.md)）
> 测试环境：192.168.13.165（Ascend 910×8，refactor 系列容器，代码目录 `/vllm-workspace/vllm-ascend`）
> 上游依据：B 修复文档遗留 P0-1（165 全量 UT 重跑）、P0-2（layerwise 指标语义实测确认）；SR 需求 not overlapped 指标验收
> 编制：2026-08-26

---

## 一、测试目标

| # | 目标 | 对应 |
|---|---|---|
| G1 | 指标分支整合到新基线（ff998aad1）后，ascend_store 全量 UT 无回归 | B 文档 P0-1 |
| G2 | layerwise 真实路径下 `load_duration_seconds{path=layerwise}` 语义正确（传输完成时刻止表，不再被计算 span 拉长） | B 文档 P0-2（M1 修复实测确认） |
| G3 | 新增 not overlapped 指标在真实 layerwise 负载下正确上报：`not_overlapped_seconds`、`overlap_ratio` 样本合理、不变式成立 | SR 需求验收项 |
| G4 | sync 路径指标回归不受 D1/D2 改动影响 | 回归确认 |

---

## 二、T0 环境核查与准备（0.5d）

165 上一次作为 UT 环境是 8 月中旬（refactor_812 容器建设期），状态需重新确认。逐项核查，全部留痕到测试报告：

| # | 核查项 | 命令/方法 | 达标 |
|---|---|---|---|
| 1 | 容器现状 | `docker ps -a`（root） | 选定基容器（优先 refactor_812；不可用则从镜像新建 `layerwise_165`） |
| 2 | vLLM 版本 | `python3 -c "import vllm; print(vllm.__version__)"` | ≥0.27.1（metrics 框架 `KVConnectorPromMetrics` 需要的版本线） |
| 3 | vllm_ascend 安装态 | `python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"` | 可导入即可，测试用源码目录跑 |
| 4 | torch_npu / NPU 卡 | `npu-smi info` | 至少 1 卡空闲（layerwise 冒烟 TP=1 用 1 卡） |
| 5 | github 连通 | `curl -s -o /dev/null -w '%{http_code}' https://github.com` | 200；不通则起反向隧道 `ssh -N -R 127.0.0.1:7897:127.0.0.1:7897 root@192.168.13.165` 后走代理 |
| 6 | 代码同步 | `git clone --depth 1 -b layerwise_pooling https://github.com/lizy124/vllm-ascend.git /root/layerwise_pooling` | HEAD 为 D5 push 后的 commit |
| 7 | **memcache 组件**（layerwise 冒烟硬前提） | `ls /usr/local/memcache_hybrid /usr/local/memfabric_hybrid` | 两者齐备则跳过安装；缺失则按 T0.1 安装 |
| 8 | **模型**（layerwise 冒烟硬前提） | `ls /mnt/weight` | layerwise 只集成 MLA（mla_v1）/SFA（sfa_v1）后端，需要 DeepSeek 系模型（如 DeepSeek-V2-Lite）；GQA 全注意力模型（qwen3 类）不触发 layerwise 逐层调用。缺失则下载 |
| 9 | pytest / ruff | `pip list \| grep -iE 'pytest\|ruff'` | pytest 可用；ruff 0.14.0（与 CI 一致） |

### T0.1 memcache 安装路径（仅第 7 项缺失时执行）

按官方文档（layerwise_kv_pool.md / layerwise_and_sparse_kv_cache_offloading.md）：

```bash
echo 200000 > /proc/sys/vm/nr_hugepages
# memfabric_hybrid release/1.2 → memcache_hybrid（gitcode 拉取，需代理）
# 验证：source set_env.sh 后 python -c "from memcache_hybrid import MetaService"
```

预计 0.5–1d。若安装受阻（源不可达/驱动不匹配），T3b 降级处理（见第六节）。

### T0.2 模型准备（仅第 8 项缺失时执行）

优先取 DeepSeek-V2-Lite（文档示例模型，MLA，TP=1 可跑，权重 ~2.4T 稀疏/16B dense 级别，下载量小）。51 服务器 `/mnt/weight` 若有 DeepSeek 系且与 165 存储互通，直接复用。

---

## 三、T1 静态检查（165，0.5h）

```bash
cd /root/layerwise_pooling
ruff check vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py \
           vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py \
           vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py \
           vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py \
           vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py \
           tests/ut/distributed/ascend_store/
ruff format --check <同上文件>
```

通过标准：无 error。mypy 在 165 可选（本地已过，CI 会兜底）。

---

## 四、T2 全量 UT（G1，0.5d）

```bash
cd /root/layerwise_pooling
PYTHONPATH=/root/layerwise_pooling python -m pytest tests/ut/distributed/ascend_store/ -v 2>&1 | tail -30
```

| 项 | 预期 |
|---|---|
| 既有用例 | 311+3 全过（B 文档口径：311 基线 + C 轮新增 1；以实跑数为准，报告记录精确值） |
| D2 新增用例 | dev_plan 第四节 6+1 个全过 |
| R1 预判用例（若提交） | 允许 fail——其输出（raise 与否）本身就是信息，单独记录 |
| 失败处置 | 任一既有用例失败 → 判定 D1 整合引入回归，回本地定位后重推 |

重点关注 `test_metrics.py` 全部、`test_pool_worker.py` 的 layerwise process 用例、`test_ascend_store_connector.py` 的 worker/scheduler 角色用例（M5 回归点）。

---

## 五、T3/T4 冒烟与指标语义验证（G2/G3/G4，1–1.5d）

### T3b layerwise 冒烟（核心，G2+G3）

**部署形态**：PD-Mixed 单实例，TP=1，1 卡。

```bash
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡号>
echo 200000 > /proc/sys/vm/nr_hugepages
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export PYTHONHASHSEED=0
# MetaService 后台起（mmc-meta.conf）

python -m vllm.entrypoints.openai.api_server \
    --model <DeepSeek-V2-Lite 路径> --port 8004 \
    --trust-remote-code --enforce-eager --no-enable-prefix-caching \
    --tensor-parallel-size 1 --max-model-len 8192 \
    --kv-transfer-config '{
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "backend": "memcache", "mooncake_rpc_port": "0", "use_layerwise": true
        }}'
```

**负载序列**（吸取 granularity=128 教训，prompt ≥ 4096 token）：

| 步骤 | 操作 | 目的 |
|---|---|---|
| S1 | 5042-token 首请求 | layerwise 逐层 save 入池 |
| S2 | 等 10s，同 prompt 复发 2–3 次 | layerwise 逐层 load，触发 wait_for_layer_load 阻塞 |
| S3 | 短 prompt 请求 1 次 | delayed_release 路径 |
| S4 | 抓取 `http://127.0.0.1:8004/metrics` | 指标快照 |

**T4 指标语义验证表**：

| # | 检查项 | 通过标准 |
|---|---|---|
| V1 | `load_duration_seconds{path=layerwise}` | 有样本；桶分布**不挤在单一桶**（B 修复前 51 实测全部挤 (0.2,0.3)s 桶、avg≈230ms≈计算 span；修复后应显著小于同请求计算耗时） |
| V2 | `load_not_overlapped_seconds{path=layerwise}` | 有样本；值 ≤ 对应 duration（不变式）；量级判断：掩盖良好时应远小于 duration |
| V3 | `load_overlap_ratio{path=layerwise}` | 样本 ∈ (0, 1]；首层暴露存在 → 严格 <1 合理 |
| V4 | 服务端聚合日志 | `KV Transfer metrics: ... not_overlapped_avg_ms=... overlap_ratio_avg=...` 字段出现且数值与 V2/V3 一致 |
| V5 | 交叉验证 | 取 2–3 个请求，日志中逐层 wait 的耗时（若 DEBUG 级可见）或事件 set 时间戳，与 not_overlapped 样本量级对照 |
| V6 | `delayed_release_requests` | S3 后 >0 或维持快照语义（最新值） |
| V7 | layerwise save 侧证据 | memcache 侧 key 数/字节数增长（有管理接口则查，无则跳过，S2 命中 load 本身即证据） |

### T3a mooncake sync 回归冒烟（可选，G4，0.5d）

条件允许时执行（165 有 mooncake master 常驻或易起）：复用 51 冒烟形态（mooncake backend、`use_layerwise=false`、5042-token 负载），验证 sync 路径 4 指标仍正常、且新增的 not_overlapped 默认语义下 sync 样本 `not_overlapped ≈ duration`、`overlap_ratio ≈ 0`。
165 无 mooncake 环境则跳过——sync 逻辑已由 UT 覆盖，51 已实测过，非本轮增量风险。

---

## 六、降级路径

| 阻塞 | 降级动作 |
|---|---|
| memcache 安装失败（T0.1） | T3b 转为 UT 级验证（mock 后端跑 layerwise 指标路径）+ 记录环境缺口；冒烟移至具备 memcache 的环境（51 或后续专用环境）补做，G2/G3 的实测确认顺延但**不阻塞** D1/D2 代码合入上游 PR 的准备 |
| DeepSeek 系模型不可得 | 用最小 MLA 模型替代；仍无则 T3b 顺延（layerwise 只支持 MLA/SFA 后端，无替代路径） |
| 165 NPU 全忙 | UT/静态照跑（CPU 即可）；T3b 排队等卡 |
| github 拉取失败 | 走反向隧道代理；仍失败则本地打包 rsync 工作树到 165 |

---

## 七、产出物与通过标准

| 产出 | 内容 |
|---|---|
| 测试报告 `test_report_165.md`（本目录） | T0 核查留痕、T1–T4 结果、指标快照原文、V1–V7 逐项判定、UT 精确计数、遗留问题清单 |
| 实验台账条目 | layerwise 冒烟的服务配置、模型、granularity 实测值，供 Phase 1 基线复用 |

**整轮通过标准**：G1 全量 UT 无回归 + G2/G3 在 T3b 实测确认（或按降级路径明确顺延并记录）。达成后 D1/D2 具备向上游指标 PR（#14912 后续）追加提交的条件。
